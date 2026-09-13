.PHONY: ui ui-install ui-serve ui-clean ui-test ui-seed

ui: ui-install ui-serve

ui-install:
	@echo "==> Installing JS dependencies"
	cd $(CURDIR) && npm install

ui-serve:
	@echo "==> Serving operator UI on http://127.0.0.1:5173"
	cd $(CURDIR) && npx vite --config vite.config.ts

ui-clean:
	rm -rf node_modules package-lock.json dist

ui-seed:
	cd $(CURDIR) && python operator_api/seed.py

ui-test: ui-seed
	cd $(CURDIR) && python test_test.py
