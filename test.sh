export ISSUE_URL="foo"
echo "s|\${{ needs.find_exercise.outputs.issue-url }}|$ISSUE_URL|g"
