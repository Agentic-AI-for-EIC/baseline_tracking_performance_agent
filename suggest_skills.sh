#!/bin/bash
session_file="${1}"
model="${2:-opencode/big-pickle}"

session_file="$(realpath "${session_file}" 2>/dev/null || echo "${session_file}")"
project_root="/home/wxie/eic/tracking_performance"

# Check required parameters before execution
if [[ -z "$session_file" ]]; then
    echo "Usage: $0 <session_file.json> [model]"
    exit 1
fi

source ~/.bashrc

# Check if session file exists
if [[ -f "$session_file" ]]; then
    session_id=$(jq -r '.info.id' "$session_file")
    echo "Found existing session file. Session ID: ${session_id}"
    
    IMPORT_CMD="opencode import ${session_file}"
    SESSION_FLAG="-s ${session_id}"
else
    echo "Session file '$session_file' not found. Starting a NEW session."
    
    IMPORT_CMD="# No session to import"
    SESSION_FLAG=""
fi

eic-shell << EOF
# Re-sequence PATH precedence inside container
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:\$PATH"
export PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v "osg-wn-client" | paste -sd:)
export PATH="\$HOME/.opencode/bin:\$HOME/eic/eic-mcp/bin:\$PATH"

# Import session file if it exists
${IMPORT_CMD}

# Execute task (resumes session if -s flag is present, otherwise starts new session)
opencode run --auto ${SESSION_FLAG} -m ${model} "Read /home/wxie/eic/tracking_performance/PLAN.md carefully. Can you check if any SKILL.md or skills are used in this project. If not, what's your suggestion of the skills to be added. Please go ahead to add the suggested skills"
EOF
