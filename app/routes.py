from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
import csv
from typing import Optional, Tuple

router = APIRouter()


class SampleRequest(BaseModel):
    sample_id: int
    value: float


def _data_csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'data' / 'sample_data.csv'


def _lookup_csv(sample_id: int) -> Tuple[Optional[float], int, Optional[float]]:
    csv_file = _data_csv_path()
    if not csv_file.exists():
        return None, 0, None
    total = 0.0
    count = 0
    found_value: Optional[float] = None
    with csv_file.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sid = int(row.get('sample_id', ''))
                val = float(row.get('value', ''))
            except (TypeError, ValueError):
                continue
            count += 1
            total += val
            if sid == sample_id:
                found_value = val
    mean = (total / count) if count else None
    return found_value, count, mean


@router.get('/health')
async def health():
    return {'status': 'ok'}


@router.post('/process')
async def process_data(request: SampleRequest):
    result = request.value * 2
    csv_value, rows, mean = _lookup_csv(request.sample_id)
    response = {
        'sample_id': request.sample_id,
        'result': result,
    }
    if csv_value is not None:
        response['csv_value'] = csv_value
    response['dataset_rows'] = rows
    if mean is not None:
        response['dataset_mean'] = mean
    return response
