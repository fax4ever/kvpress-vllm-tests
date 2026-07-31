#!/usr/bin/env bash
set -euo pipefail

fmt_duration() {
    local secs=$1
    local h=$((secs / 3600))
    local m=$(( (secs % 3600) / 60 ))
    local s=$((secs % 60))
    if [[ $h -gt 0 ]]; then
        printf "%dh%02dm%02ds" $h $m $s
    elif [[ $m -gt 0 ]]; then
        printf "%dm%02ds" $m $s
    else
        printf "%ds" $s
    fi
}

max_failures=0
timeout_val=""
tb_style="short"
verbose=""
collect_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -x)
            max_failures=1
            shift
            ;;
        --max-failures=*)
            max_failures="${1#--max-failures=}"
            shift
            ;;
        --timeout=*)
            timeout_val="${1#--timeout=}"
            shift
            ;;
        --tb=*)
            tb_style="${1#--tb=}"
            shift
            ;;
        -v|--verbose)
            verbose="-v"
            shift
            ;;
        --forked)
            shift
            ;;
        *)
            collect_args+=("$1")
            shift
            ;;
    esac
done

echo "=== Collecting tests ==="
test_ids=()
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" != *::* ]] && continue
    test_ids+=("$line")
done < <(pytest --collect-only -q "${collect_args[@]}" 2>/dev/null || true)

total=${#test_ids[@]}
echo "=== Found $total tests ==="

if [[ $total -eq 0 ]]; then
    echo "No tests collected."
    exit 0
fi

passed=0
failed=0
skipped=0
errors=0
failed_tests=()

run_args=()
[[ -n "$verbose" ]] && run_args+=("$verbose")
run_args+=("--tb=$tb_style")
[[ -n "$timeout_val" ]] && run_args+=("--timeout=$timeout_val")
run_args+=("--no-header" "-q")

suite_start=$SECONDS

for i in "${!test_ids[@]}"; do
    test_id="${test_ids[$i]}"
    n=$((i + 1))
    test_start=$SECONDS
    echo ""
    so_far=$(fmt_duration $((SECONDS - suite_start)))
    echo "[$n/$total | $so_far] $test_id"

    set +e
    output=$(pytest "${run_args[@]}" "$test_id" 2>&1)
    rc=$?
    set -e

    elapsed=$((SECONDS - test_start))
    elapsed_fmt=$(fmt_duration $elapsed)

    if echo "$output" | grep -q "SKIPPED\|skipped"; then
        echo "  SKIPPED ($elapsed_fmt)"
        skipped=$((skipped + 1))
    elif [[ $rc -eq 0 ]]; then
        echo "  PASSED ($elapsed_fmt)"
        passed=$((passed + 1))
    else
        echo "  FAILED ($elapsed_fmt)"
        echo "$output"
        failed=$((failed + 1))
        failed_tests+=("$test_id")
        if [[ $max_failures -gt 0 && $failed -ge $max_failures ]]; then
            echo ""
            echo "=== Stopping after $max_failures failure(s) ==="
            break
        fi
    fi
done

total_elapsed=$((SECONDS - suite_start))
total_fmt=$(fmt_duration $total_elapsed)

echo ""
echo "============================== summary =============================="
echo "$passed passed, $failed failed, $skipped skipped, $errors errors (out of $total) in $total_fmt"
if [[ ${#failed_tests[@]} -gt 0 ]]; then
    echo ""
    echo "Failed tests:"
    for t in "${failed_tests[@]}"; do
        echo "  FAILED $t"
    done
fi

[[ $failed -gt 0 ]] && exit 1
exit 0
