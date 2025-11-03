File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
File "C:\Users\p90022569\Downloads\score\venv\lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 669, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
File "C:\Users\p90022569\Downloads\score\app.py", line 691, in <module>
    guiding_principle_points = score_guiding_principle(parsed if isinstance(parsed, dict) else {}, abstracts)
File "C:\Users\p90022569\Downloads\score\app.py", line 188, in score_guiding_principle
    fulltext = " ".join(abstracts).lower()
