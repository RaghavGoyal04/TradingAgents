"""Local, read-only Streamlit dashboard for the portfolio intelligence run.

The UI only reads run artifacts and launches/resumes the orchestrator. It never
executes analysis in-process. Run it bound to localhost:

    streamlit run dashboard/app.py --server.address 127.0.0.1
"""
