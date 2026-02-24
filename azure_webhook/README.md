# Azure Functions Google Drive Webhook

This directory contains an Azure Function acting as a webhook listener for Google Drive Push Notifications. The project uses `uv` for dependency management to keep requirements localized and avoid polluting the parent repository.

## 1. Google Setup

1. **Service Account:**
   * Go to the [Google Cloud Console](https://console.cloud.google.com/).
   * Enable the **Google Drive API**.
   * Create a Service Account and download the JSON key.
2. **Drive Folder Sharing:**
   * Find the Google Drive Folder ID you want to watch.
   * Share this folder with the Service Account email address as a *Viewer* or *Editor*.

## 2. Dependency Management & Azure Build prep

Because this uses `uv`, we must generate a `requirements.txt` file before deploying to Azure (since Azure's Python worker looks for it during the build step).

From within the `azure_webhook` directory run:
```bash
uv export --format requirements-txt > requirements.txt
```

If you ever need to add new packages, run:
```bash
uv add <package_name>
uv export --format requirements-txt > requirements.txt
```

## 3. Azure Deployment Settings

After deploying to Azure Functions (Flex Consumption or standard Consumption), configure the following **Application Settings (Environment Variables)** in the Azure Portal:

| Variable Name | Description |
|---|---|
| `GDRIVE_TARGET_DIR` | The Google Drive Folder ID to watch. |
| `GDRIVE_WEBHOOK_TOKEN` | A secret token you select. Must match the value sent during watch subscription. |
| `WEBHOOK_DOMAIN` | The domain of your Azure app (e.g. `myapp.azurewebsites.net`). |
| `GITHUB_TOKEN` | A GitHub Personal Access Token with permissions to trigger `workflow_dispatch` actions. |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | The raw contents of your Service Account JSON file. Flatten it into a single line string. |

*(Note: Locally, these variables can safely fall back to the `../config.toml` and `../token.json` setup, making local testing on a VM very simple.)*

## 4. Manual Channel Refreshing

Google Drive Push Notification channels expire every ~24 hours. The included Azure Timer function `refresh_channel_timer` will handle this automatically by executing every 23 hours.

If you are just running locally on a VM and need to manually refresh the channel, you can run:

```bash
uv run python refresh_channel.py
```
This stops the old channel (via `state.json`) and starts a new one with a fresh UUID.
