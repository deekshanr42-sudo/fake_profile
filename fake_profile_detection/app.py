"""
=============================================================================
  FAKE PROFILE DETECTION — FLASK WEB APPLICATION
  ──────────────────────────────────────────────────────────
  This is the classic HTML/CSS/JS frontend backed by Flask.
=============================================================================
"""

from server import app

if __name__ == '__main__':
    app.run(debug=True, port=5000)
