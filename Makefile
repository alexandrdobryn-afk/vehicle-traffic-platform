.PHONY: up down build logs shell-backend shell-training migrate models trt clean health

# ── Start ──────────────────────────────────────────────────────────
up:
	bash scripts/start.sh --foreground
up-d:
	bash scripts/start.sh
up-monitor:
	bash scripts/start.sh --profile monitoring

# ── Build ──────────────────────────────────────────────────────────
build:
	docker compose build --no-cache
build-training:
	docker compose build --no-cache training-api training-worker

# ── Stop ───────────────────────────────────────────────────────────
down:
	docker compose down
down-v:
	docker compose down -v

# ── Logs ───────────────────────────────────────────────────────────
logs:
	docker compose logs -f
logs-backend:
	docker compose logs -f backend
logs-training:
	docker compose logs -f training-api
logs-worker:
	docker compose logs -f training-worker

# ── Shell ──────────────────────────────────────────────────────────
shell-backend:
	docker compose exec backend bash
shell-frontend:
	docker compose exec frontend sh
shell-training:
	docker compose exec training-api bash
shell-worker:
	docker compose exec training-worker bash
shell-db:
	docker compose exec postgres psql -U vtp_user -d vtp_db

# ── Database ───────────────────────────────────────────────────────
migrate:
	docker compose exec backend alembic upgrade head
migrate-training:
	docker compose exec training-api alembic upgrade head
migrate-all: migrate migrate-training
migrate-create:
	docker compose exec backend alembic revision --autogenerate -m "$(name)"
migrate-training-create:
	docker compose exec training-api alembic revision --autogenerate -m "$(name)"

# ── Models ─────────────────────────────────────────────────────────
models:
	cd scripts && pip install ultralytics paddleocr cryptography -q && python download_models.py
trt:
	cd scripts && python export_tensorrt.py --target all --size n --half

# ── Dev (without docker) ───────────────────────────────────────────
dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000
dev-training:
	cd training/backend && uvicorn app.main:app --reload --port 8001
dev-worker:
	cd training/backend && celery -A app.celery_app worker --loglevel=info --queues=gpu,cpu --concurrency=1
dev-frontend:
	cd frontend && npm run dev

# ── Flower ─────────────────────────────────────────────────────────
flower:
	docker compose --profile monitoring up flower -d
	@echo "Flower UI: http://localhost:5555"

# ── Cleanup ────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ── Health ─────────────────────────────────────────────────────────
health:
	@echo "=== Inference Backend ===" && curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
	@echo "\n=== Training API ===" && curl -s http://localhost:8001/api/v1/training/health | python3 -m json.tool

# ── Keys ───────────────────────────────────────────────────────────
fernet-key:
	python3 -c "from cryptography.fernet import Fernet; print('FERNET_KEY=' + Fernet.generate_key().decode())"
secret-key:
	python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# ── Full setup from scratch ────────────────────────────────────────
setup:
	@echo "Step 1: Generate keys"
	@python3 -c "from cryptography.fernet import Fernet; import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32)); print('FERNET_KEY=' + Fernet.generate_key().decode())"
	@echo "\nStep 2: Copy keys above to .env, then run:"
	@echo "  make models && make build && make up-d && make migrate-all"
