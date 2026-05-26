"""
Client minimale per Betfair Exchange API — SOLA LETTURA (odds live tennis).
NON piazza scommesse (paper-trading). Per scommettere davvero servirebbe placeOrders,
volutamente non implementato.

Setup (una volta):
  1. Account Betfair + iscrizione al Developer Program -> ottieni un Application Key:
     https://developer.betfair.com/  (chiave "delayed" gratuita per i dati)
  2. Esporta le credenziali come variabili d'ambiente (NON metterle nel codice):
       export BF_APP_KEY="la-tua-app-key"
       export BF_USERNAME="email-betfair"
       export BF_PASSWORD="password-betfair"
  3. Exchange italiano: endpoint identitysso.betfair.it (default sotto).

Uso:
    from betfair_client import BetfairClient
    bf = BetfairClient(); bf.login()
    for m in bf.list_inplay_tennis():
        print(m["event"], m["market_id"])
        print(bf.best_back_prices(m["market_id"]))
"""

from __future__ import annotations

import os

import requests

LOGIN_URL = os.environ.get("BF_LOGIN_URL", "https://identitysso.betfair.it/api/login")
BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
TENNIS_EVENT_TYPE_ID = "2"


class BetfairClient:
    def __init__(self, app_key=None, username=None, password=None):
        self.app_key = app_key or os.environ.get("BF_APP_KEY")
        self.username = username or os.environ.get("BF_USERNAME")
        self.password = password or os.environ.get("BF_PASSWORD")
        self.token = None
        if not all([self.app_key, self.username, self.password]):
            raise RuntimeError(
                "Credenziali mancanti. Esporta BF_APP_KEY, BF_USERNAME, BF_PASSWORD "
                "(vedi docstring di betfair_client.py)."
            )

    def login(self) -> str:
        r = requests.post(
            LOGIN_URL,
            data={"username": self.username, "password": self.password},
            headers={"X-Application": self.app_key, "Accept": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        js = r.json()
        if js.get("status") != "SUCCESS":
            raise RuntimeError(f"Login Betfair fallito: {js}")
        self.token = js["token"]
        return self.token

    def _rpc(self, method: str, params: dict):
        if not self.token:
            raise RuntimeError("Esegui login() prima.")
        payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}",
                   "params": params, "id": 1}
        r = requests.post(
            BETTING_URL, json=payload,
            headers={"X-Application": self.app_key, "X-Authentication": self.token,
                     "content-type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            raise RuntimeError(f"Betfair API error: {js['error']}")
        return js["result"]

    def list_inplay_tennis(self) -> list[dict]:
        """Mercati MATCH_ODDS del tennis attualmente IN-PLAY."""
        res = self._rpc("listMarketCatalogue", {
            "filter": {"eventTypeIds": [TENNIS_EVENT_TYPE_ID],
                       "inPlayOnly": True,
                       "marketTypeCodes": ["MATCH_ODDS"]},
            "maxResults": 100,
            "marketProjection": ["EVENT", "RUNNER_DESCRIPTION"],
        })
        out = []
        for m in res:
            out.append({
                "market_id": m["marketId"],
                "event": m.get("event", {}).get("name", "?"),
                "runners": [{"selection_id": r["selectionId"], "name": r["runnerName"]}
                            for r in m.get("runners", [])],
            })
        return out

    def best_back_prices(self, market_id: str) -> dict:
        """selection_id -> miglior quota back disponibile (e size)."""
        res = self._rpc("listMarketBook", {
            "marketIds": [market_id],
            "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
        })
        if not res:
            return {}
        out = {}
        for r in res[0].get("runners", []):
            backs = r.get("ex", {}).get("availableToBack", [])
            if backs:
                out[r["selectionId"]] = {"price": backs[0]["price"], "size": backs[0]["size"]}
        return out

    def place_bet(self, *a, **k):
        raise NotImplementedError(
            "Piazzamento scommesse disabilitato di proposito: questo client e' "
            "paper-trading. Implementa placeOrders solo quando l'edge e' provato."
        )
