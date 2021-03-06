"""
This script runs the Foodology application using a development server.
"""

from os import environ
from terror_ai import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '1111'))
    except ValueError:
        PORT = 2222
    app.run(HOST, PORT, debug=True)
