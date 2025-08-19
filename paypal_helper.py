import os
import base64
import json
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET")
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox")

if PAYPAL_ENV == "live":
    PAYPAL_BASE = "https://api-m.paypal.com"
else:
    PAYPAL_BASE = "https://api-m.sandbox.paypal.com"


class PayPalError(Exception):
    pass


def _get_basic_auth_header() -> Dict[str, str]:
    if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
        raise PayPalError("PayPal credentials not set in environment")
    token = f"{PAYPAL_CLIENT_ID}:{PAYPAL_SECRET}".encode()
    b64 = base64.b64encode(token).decode()
    return {"Authorization": f"Basic {b64}", "Content-Type": "application/x-www-form-urlencoded"}


def get_access_token() -> str:
    """Obtain an OAuth2 access token from PayPal."""
    url = f"{PAYPAL_BASE}/v1/oauth2/token"
    headers = _get_basic_auth_header()
    resp = requests.post(url, headers=headers, data={"grant_type": "client_credentials"}, timeout=10)
    if resp.status_code != 200:
        raise PayPalError(f"Failed to obtain access token: {resp.status_code} {resp.text}")
    data = resp.json()
    return data.get("access_token")


def create_order(amount: str, currency: str = "USD", return_url: Optional[str] = None, cancel_url: Optional[str] = None) -> Dict[str, Any]:
    """Create a PayPal order and return approval url and order id.

    amount: numeric string like '10.00'
    """
    access_token = get_access_token()
    url = f"{PAYPAL_BASE}/v2/checkout/orders"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [{
            "amount": {"currency_code": currency, "value": amount}
        }]
    }
    if return_url or cancel_url:
        payload["application_context"] = {}
        if return_url:
            payload["application_context"]["return_url"] = return_url
        if cancel_url:
            payload["application_context"]["cancel_url"] = cancel_url

    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    if resp.status_code not in (201, 200):
        raise PayPalError(f"Failed to create order: {resp.status_code} {resp.text}")
    data = resp.json()
    # find approval link
    approval_link = None
    for link in data.get("links", []):
        if link.get("rel") == "approve":
            approval_link = link.get("href")
            break
    return {"order_id": data.get("id"), "approval_link": approval_link, "raw": data}


def capture_order(order_id: str) -> Dict[str, Any]:
    """Capture funds for an approved order."""
    access_token = get_access_token()
    url = f"{PAYPAL_BASE}/v2/checkout/orders/{order_id}/capture"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={}, timeout=10)
    if resp.status_code not in (200, 201):
        raise PayPalError(f"Failed to capture order: {resp.status_code} {resp.text}")
    return resp.json()


def verify_webhook_signature(transmission_id: str, timestamp: str, webhook_id: str, event_body: Dict[str, Any], cert_url: str, auth_algo: str, transmission_sig: str) -> bool:
    """Basic wrapper for PayPal's verify-webhook-signature endpoint.

    See: https://developer.paypal.com/docs/api/webhooks/v1/#verify-webhook-signature_post
    This function requires the same webhook_id you registered in the PayPal dashboard.
    """
    access_token = get_access_token()
    url = f"{PAYPAL_BASE}/v1/notifications/verify-webhook-signature"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {
        "transmission_id": transmission_id,
        "transmission_time": timestamp,
        "cert_url": cert_url,
        "auth_algo": auth_algo,
        "transmission_sig": transmission_sig,
        "webhook_id": webhook_id,
        "webhook_event": event_body,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    if resp.status_code != 200:
        raise PayPalError(f"Webhook signature verification failed: {resp.status_code} {resp.text}")
    data = resp.json()
    return data.get("verification_status") == "SUCCESS"
