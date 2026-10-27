# Sourced by jobs and interactive shells: puts the venv and model cache in the course
# student dir (home quota is small, and course storage is 1 TB shared).
# Set COURSE_DIR once if the auto-detect picks the wrong term.
: "${COURSE_DIR:=$(ls -d /courses/CS5180.*/students/"$USER" 2>/dev/null | tail -1)}"
if [ -z "$COURSE_DIR" ]; then
    echo "no /courses/CS5180.*/students/$USER; set COURSE_DIR" >&2
    return 1 2>/dev/null || exit 1
fi
export COURSE_DIR
export HF_HOME="$COURSE_DIR/hf"
source "$COURSE_DIR/venv/bin/activate"
