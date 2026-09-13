# AUTONOMOUS LOOP WRAPPER
# Runs each agent in a loop until TASKS.md has no unchecked items
# Then writes FINAL_REPORT.md and stops

\ = 20
\ = 0

while (\ -lt \) {
    \++
    Write-Host "=== LOOP ITERATION \ / \ ==="

    # Check if all tasks are done
    \ = (Get-Content TASKS.md | Select-String '^\- \[ \]').Count
    if (\ -eq 0) {
        Write-Host "ALL TASKS COMPLETE. Writing FINAL_REPORT.md..."
        @"
# FINAL REPORT — All Tasks Complete
Generated: \09/12/2026 14:56:00
Iterations: \
