"""
google_auth.py

One OAuth login shared by the Gmail and Calendar clients.

Setup (once):
  1. Put the OAuth client file you download from Google Cloud Console at
     integrations/credentials.json   (type: Desktop app)
  2. From the repo root run:   python -m integrations.google_auth
     A browser window opens, you approve access, and integrations/token.json
     is saved. The API server never opens a browser itself -- it only loads
     the saved token and refreshes it when needed.

Scopes are deliberately minimal:
  - gmail.readonly   -> read mail only (cannot send, delete or modify)
  - calendar.events  -> create/read events only (not full calendar admin)

If you ever change SCOPES, delete token.json and log in again.

While the OAuth consent screen is in "Testing" mode, Google expires the
refresh token after 7 days -- you'll just need to run the login command again.

credentials.json and token.json are secrets: both are in .gitignore.
"""

import os
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

LOGIN_HINT = "Google login needed. From the repo root run: python -m integrations.google_auth"


class NotAuthorized(Exception):
    """No valid saved login. The message tells the user how to fix it."""


def _save(creds: Credentials) -> None:
    TOKEN_FILE.write_text(creds.to_json())
    try:
        os.chmod(TOKEN_FILE, 0o600)  # best effort; no-op on Windows
    except OSError:
        pass


def get_credentials(interactive: bool = False) -> Credentials:
    """
    Returns valid credentials. With interactive=False (what the API server
    uses) this raises NotAuthorized instead of opening a browser.
    """
    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except ValueError:
            creds = None  # corrupt token file

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save(creds)
            return creds
        except RefreshError:
            creds = None  # refresh token expired/revoked -> log in again

    if not interactive:
        raise NotAuthorized(LOGIN_HINT)

    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"{CREDENTIALS_FILE} not found. Download the OAuth client JSON "
            "(Desktop app) from Google Cloud Console and save it there."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    _save(creds)
    return creds


if __name__ == "__main__":
    from googleapiclient.discovery import build

    creds = get_credentials(interactive=True)
    profile = build("gmail", "v1", credentials=creds, cache_discovery=False) \
        .users().getProfile(userId="me").execute()
    print(f"Authorized as {profile['emailAddress']}. Token saved to {TOKEN_FILE}")