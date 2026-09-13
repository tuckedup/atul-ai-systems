#!/bin/sh
set -e
cd "$(dirname "$0")"
npm run dev -- --host 127.0.0.1
