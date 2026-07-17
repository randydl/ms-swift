#!/bin/bash

fkill() {
    kw="$1"; u="${2:-$(whoami)}"
    [ -z "$kw" ] && echo "Usage: fkill <kw> [user]" && return
    psu=$(ps -u "$u" -o pid,cmd --no-header | grep "$kw" | grep -v grep)
    [ -z "$psu" ] && echo "No match" && return
    echo "$psu" | tee /dev/tty | awk '{print $1}' | xargs kill -9 && echo "Done"
}

nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | while read -r pid; do
    cmd=$(ps -p "$pid" -o args= 2>/dev/null)

    if [[ "$cmd" == *"examples/llava_ov_1_5/pretrain.py"* ]]; then
        echo "Killing PID: $pid"
        echo "$cmd"
        sudo kill -9 "$pid"
    fi
done

fkill 'swift/cli/_megatron/sft.py'
