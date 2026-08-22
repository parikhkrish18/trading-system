FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
# Explicit, deliberate non-loopback bind: a container has to listen on
# 0.0.0.0 for its published port to work. The token check in server.py
# reads this same variable, so a networked dashboard without
# DASHBOARD_API_TOKEN set refuses its mutating endpoints.
ENV DASHBOARD_HOST=0.0.0.0
CMD ["python", "-m", "monitoring.dashboard.server"]
