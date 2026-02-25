import asyncio
import json
import logging
import os
import tomllib
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, Request, Response
import requests
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from moon_reader import compute_read_stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_config():
    """Fallback to config.toml if running locally."""
    config = {}
    config_file = Path(__file__).parent.parent / "config.toml"
    if config_file.exists():
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
            gdrive = data.get("gdrive", {})
            github = data.get("github", {})
            config["target_dir"] = gdrive.get("target_dir")
            config["webhook_token"] = gdrive.get("webhook_token")
            config["webhook_url"] = gdrive.get("webhook_url")
            config["sa_json"] = gdrive.get("sa_json")
            config["github_workflow_url"] = github.get("workflow_url")
            config["github_token"] = github.get("token")

    # Environment variables take precedence
    target_dir = os.environ.get("GDRIVE_TARGET_DIR", config.get("target_dir"))
    webhook_token = os.environ.get("GDRIVE_WEBHOOK_TOKEN", config.get("webhook_token"))
    webhook_url = os.environ.get("GDRIVE_WEBHOOK_URL", config.get("webhook_url"))
    github_token = os.environ.get("GITHUB_TOKEN", config.get("github_token"))
    github_workflow_url = os.environ.get(
        "GITHUB_WORKFLOW_URL", config.get("github_workflow_url")
    )
    sa_json = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS_JSON", config.get("sa_json")
    )

    return {
        "target_dir": target_dir,
        "webhook_token": webhook_token,
        "webhook_url": webhook_url,
        "github_token": github_token,
        "github_workflow_url": github_workflow_url,
        "sa_json": sa_json,
    }


def get_drive_service(config):
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = None

    if config["sa_json"]:
        if config["sa_json"].strip().startswith("{"):
            sa_info = json.loads(config["sa_json"])
        else:
            with open(config["sa_json"], "r") as f:
                sa_info = json.load(f)
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=scopes
        )
    else:
        token_path = Path(__file__).parent.parent / "token.json"
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)
        else:
            raise RuntimeError("No Google credentials available.")

    return build("drive", "v3", credentials=creds)


def download_file(service, file_id: str) -> bytes:
    request = service.files().get_media(fileId=file_id)
    file_buf = BytesIO()
    downloader = MediaIoBaseDownload(file_buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return file_buf.getvalue()


def trigger_github_actions_sync(file_content: str, token: str, workflow_url: str):
    if not token or not workflow_url:
        logging.warning(
            "GITHUB_TOKEN or GITHUB_WORKFLOW_URL not configured. Skipping GitHub Actions trigger."
        )
        return

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    payload = {"ref": "master", "inputs": {"read_stats": file_content}}

    resp = requests.post(workflow_url, headers=headers, json=payload)
    if resp.status_code >= 400:
        logging.error(f"Failed to trigger GH action: {resp.status_code} {resp.text}")
    else:
        logging.info("Successfully triggered GitHub Actions.")


def process_latest_file_sync(config):
    service = get_drive_service(config)
    target_dir = config["target_dir"]

    results = (
        service.files()
        .list(
            q=f"'{target_dir}' in parents",
            orderBy="modifiedTime desc",
            pageSize=1,
            fields="files(id, name, createdTime, modifiedTime)",
            spaces="drive",
        )
        .execute()
    )

    items = results.get("files", [])
    if not items:
        logging.info("No files found in target folder.")
        return "No files found"

    latest_file = items[0]
    logging.info(f"Latest file identified: {latest_file['name']} ({latest_file['id']})")

    content_bytes = download_file(service, latest_file["id"])

    try:
        process_result = compute_read_stats.process_archive(content_bytes)
    except Exception as e:
        logging.error(f"Failed to process data relative to compute_read_stats: {e}")
        raise RuntimeError(f"Processing error: {str(e)}")

    trigger_github_actions_sync(
        process_result,
        config.get("github_token"),
        config.get("github_workflow_url"),
    )
    return "Webhook processed successfully"


def stop_channel(config):
    service = get_drive_service(config)
    state_file = Path(__file__).parent / "state.json"

    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            if "id" in state and "resourceId" in state:
                service.channels().stop(
                    body={"id": state["id"], "resourceId": state["resourceId"]}
                ).execute()
                logging.info("Stopped old channel successfully.")
        except Exception as e:
            logging.warning(
                f"Could not stop old channel (might be already expired): {e}"
            )

        try:
            state_file.unlink()
        except OSError:
            pass
        return True

    logging.info("No channel state found to stop.")
    return False


def perform_channel_refresh(config):
    service = get_drive_service(config)
    target_dir = config["target_dir"]
    webhook_url = config["webhook_url"]
    webhook_token = config["webhook_token"]

    if not webhook_url:
        logging.error("No webhook_url configured.")
        return

    state_file = Path(__file__).parent / "state.json"

    stop_channel(config)

    new_uuid = str(uuid.uuid4())
    expiration_ms = int(datetime.now(timezone.utc).timestamp() * 1000) + (
        24 * 60 * 60 * 1000
    )

    resp = (
        service.files()
        .watch(
            fileId=target_dir,
            body={
                "token": webhook_token,
                "id": new_uuid,
                "type": "webhook",
                "expiration": str(expiration_ms),
                "address": webhook_url,
            },
        )
        .execute()
    )

    logging.info(f"Successfully started new watch channel: {resp['id']}")

    with open(state_file, "w") as f:
        json.dump(resp, f)


async def channel_refresh_loop():
    """Background task to refresh the channel roughly every 22 hours."""
    while True:
        await asyncio.sleep(22 * 60 * 60)
        config = get_config()
        try:
            logging.info("Running scheduled channel refresh.")
            await asyncio.to_thread(perform_channel_refresh, config)
        except Exception as e:
            logging.error(f"Failed to refresh channel via loop: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting up FastAPI application")
    loop_task = asyncio.create_task(channel_refresh_loop())
    yield
    # Shutdown
    logging.info("Shutting down FastAPI application")
    loop_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/google_drive_webhook")
async def google_drive_webhook(req: Request) -> Response:
    config = get_config()

    req_token = req.headers.get("x-goog-channel-token")
    if not req_token or req_token != config.get("webhook_token"):
        logging.error("Unauthorized webhook request: invalid token.")
        return Response(content="Not authorized", status_code=401)

    resource_state = req.headers.get("x-goog-resource-state")
    if resource_state == "sync":
        logging.info("Received sync message. Successfully subscribed.")
        return Response(content="OK", status_code=200)

    changed = req.headers.get("x-goog-changed", "")
    if resource_state != "update" or "children" not in changed:
        logging.info(
            f"Ignoring irrelevant event: state={resource_state}, changed={changed}"
        )
        return Response(content="Event ignored", status_code=200)

    logging.info("Processing file update")
    try:
        result_msg = await asyncio.to_thread(process_latest_file_sync, config)
        return Response(content=result_msg, status_code=200)
    except Exception as e:
        logging.error(f"Error processing webhook: {e}")
        return Response(content="Server Error", status_code=500)


@app.post("/api/refresh_channel")
async def manual_refresh_channel() -> Response:
    """Manual trigger to refresh the Google Drive watch channel."""
    config = get_config()
    try:
        await asyncio.to_thread(perform_channel_refresh, config)
        return Response(content="Successfully refreshed channel", status_code=200)
    except Exception as e:
        logging.error(f"Failed to refresh channel manually: {e}")
        return Response(content=f"Error: {e}", status_code=500)


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Google Drive Watcher FastAPI service."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the service on"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Port to run the service on"
    )
    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
