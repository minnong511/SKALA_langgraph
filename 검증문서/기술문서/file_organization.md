# 파일 정리 이력

초기 루트의 Python 파일 15개는 `src/legacy/`로 내용 변경 없이 이동했습니다.
초기 `READMD.md`는 `검증문서/기술문서/archive/READMD.md`에 보존했습니다.
이후 Notion 명세 전체 구현 요청에 따라 `src/`의 구조용 파일을 실제 구현으로 채웠습니다.
현재 배치는 [디렉토리 구조](directory_structure.md), 실행 방법은 [README](../../README.md)를 참조합니다.
`src/legacy/`는 원본 보관용이며 신규 구현에서는 import하지 않습니다.

기술문서, 디버깅 이력, 테스트 기록은 `검증문서/`로 통합했습니다.
기존 API 점검 코드는 `scripts/api_smoke.py`로 이동했으며 결과는 `검증문서/테스트/실행결과/`에 저장합니다.
