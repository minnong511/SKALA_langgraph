"""모의 테스트에서는 외부 LangSmith 전송을 수행하지 않는다.

실제 보고서 실행은 이 파일을 거치지 않으므로 .env의 LANGSMITH_TRACING=true가 유지된다.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
# 사용자가 .env에 실키를 넣어도 단위 테스트가 유료 API를 호출하지 않도록 한다.
# 실제 LLM/검색이 필요한 테스트는 호출 함수 자체를 Mock으로 주입한다.
os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""
