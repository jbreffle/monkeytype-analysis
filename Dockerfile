FROM python:3.10

# Install dependencies in a single layer and clean up in the same RUN to reduce image size
RUN python -m pip install --upgrade pip && \
    pip install flake8 pytest && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# First, copy only requirements.txt and install Python dependencies to leverage Docker cache
COPY requirements.txt .
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Then copy the rest of the application
COPY . .
