ISSUE_URL="https://github.com/Night-Hawkeye/skills-introduction-to-github/issues/286"
sed -i "s|\\\${{ needs.find_exercise.outputs.issue-url }}|$ISSUE_URL|g" test_sed_target.txt
