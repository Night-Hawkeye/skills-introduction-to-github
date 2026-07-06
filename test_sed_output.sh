ISSUE_URL="https://github.com/Night-Hawkeye/skills-introduction-to-github/issues/286"
export ISSUE_URL
export TARGET='${{ needs.find_exercise.outputs.issue-url }}'
python3 -c "import os; f='test_sed_target.txt'; c=open(f).read(); open(f,'w').write(c.replace(os.environ['TARGET'], os.environ['ISSUE_URL']))"
