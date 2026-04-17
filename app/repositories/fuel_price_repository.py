from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html as html_lib
import re
from typing import Protocol

import httpx


@dataclass
class FuelPriceInfo:
    petrol_rub_per_liter: float
    diesel_rub_per_liter: float
    source: str
    source_url: str | None
    price_date: str | None
    retrieved_at: datetime


class FuelPriceRepository(Protocol):
    async def fetch(self) -> FuelPriceInfo:
        raise NotImplementedError


class RosstatFuelPriceRepository(FuelPriceRepository):
    def __init__(self, source_url: str, timeout_seconds: float = 10.0) -> None:
        self._source_url = source_url
        self._timeout_seconds = timeout_seconds

    async def fetch(self) -> FuelPriceInfo:
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(self._source_url)

        response.raise_for_status()

        normalized = self._normalize_html(response.text)
        petrol, diesel = self._extract_prices(normalized)
        price_date = self._extract_price_date(normalized)

        return FuelPriceInfo(
            petrol_rub_per_liter=petrol,
            diesel_rub_per_liter=diesel,
            source="Росстат",
            source_url=self._source_url,
            price_date=price_date,
            retrieved_at=datetime.now(timezone.utc),
        )

    def _normalize_html(self, html: str) -> str:
        text = html_lib.unescape(html)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_prices(self, text: str) -> tuple[float, float]:
        # Expected Rosstat table row for "Российская Федерация" with prices:
        # бензин (средняя), АИ-92, АИ-95, АИ-98+, дизель
        rf_match = re.search(
            r"Российская Федерация\*?\s+(\d+[.,]\d+)\s+(\d+[.,]\d+)\s+(\d+[.,]\d+)\s+(\d+[.,]\d+)\s+(\d+[.,]\d+)",
            text,
        )
        if rf_match:
            petrol = self._to_float(rf_match.group(1))
            diesel = self._to_float(rf_match.group(5))
            return petrol, diesel

        # Fallback: try to locate "Бензин автомобильный" and "Дизельное топливо" nearby.
        petrol_match = re.search(r"Бензин автомобильный\s+(\d+[.,]\d+)", text)
        diesel_match = re.search(r"Дизельное топливо\s+(\d+[.,]\d+)", text)
        if petrol_match and diesel_match:
            return self._to_float(petrol_match.group(1)), self._to_float(diesel_match.group(1))

        raise ValueError("Unable to parse fuel prices from Rosstat page.")

    def _extract_price_date(self, text: str) -> str | None:
        date_match = re.search(r"на\s+(\d{1,2}\s+[А-Яа-яё]+\s+20\d{2})\s+года", text)
        if date_match:
            return date_match.group(1)
        return None

    @staticmethod
    def _to_float(value: str) -> float:
        return float(value.replace(",", "."))
