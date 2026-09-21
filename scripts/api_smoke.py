"""Live API checks; persist only status metadata, never credentials or response bodies.

Run from the repository root: python scripts/api_smoke.py
"""

import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

from src.agents.base import ResearchOutput  # noqa: E402
from src.config import Settings, create_llm  # noqa: E402
from src.schemas import AgentContext  # noqa: E402
from src.tools.web_search import TavilySearch  # noqa: E402


def main():
    results = []

    def check(name, operation):
        start = monotonic()
        record = {'check': name}
        try:
            assert operation(), 'Empty or unexpected response'
            record['status'] = 'PASS'
        except Exception as exc:
            record.update(status='FAIL', error_type=type(exc).__name__)
            response = getattr(exc, 'response', None)
            cause = exc.__cause__
            if response is None and cause is not None:
                response = getattr(cause, 'response', None)
            status = getattr(exc, 'status_code', None) or getattr(response, 'status_code', None)
            if isinstance(status, int):
                record['http_status'] = status
            # Whitelist diagnostic categories instead of logging arbitrary exception text.
            body = str(getattr(exc, 'body', '')).lower()
            if 'invalid model' in body or 'model_not_found' in body:
                record['category'] = 'invalid_model_id'
            elif status in (401, 403):
                record['category'] = 'authentication_or_access'
            elif status == 429:
                record['category'] = 'quota_or_rate_limit'
        record['seconds'] = round(monotonic() - start, 2)
        results.append(record)
        print(json.dumps(record), flush=True)
        return record['status'] == 'PASS'

    settings = None

    def configure():
        nonlocal settings
        settings = Settings.from_env(ROOT / '.env', llm_timeout_seconds=45)
        settings.validate_live()
        return True

    if check('configuration', configure):
        def text_call():
            return bool(create_llm(settings).invoke('Reply with only OK.').content)

        def structured_call():
            context = AgentContext(llm=create_llm(settings), structured_output_method='function_calling')
            result = context.ask(
                ResearchOutput,
                'API schema test. Return summary OK and empty lists for all other fields. Do not research.',
                {'test': 'schema validation'},
            )
            return isinstance(result, ResearchOutput) and result.summary.strip() == 'OK'

        check('llm_text', text_call)
        check('llm_research_output_schema', structured_call)
        check('tavily_search', lambda: bool(TavilySearch(
            settings.tavily_api_key.get_secret_value(), timeout_seconds=20, search_depth='basic',
        ).search('CXL memory', limit=1)))

    now = datetime.now(timezone.utc)
    report = {'timestamp_utc': now.isoformat(), 'python': platform.python_version(), 'results': results}
    directory = ROOT / '검증문서' / '테스트' / '실행결과'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (now.strftime('%Y%m%dT%H%M%S%fZ') + '-api.json')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('Report: ' + str(destination.relative_to(ROOT)))
    return 0 if all(row['status'] == 'PASS' for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
