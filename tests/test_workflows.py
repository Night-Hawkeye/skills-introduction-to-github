import yaml
import pytest
import os

def load_workflow(filename):
    filepath = os.path.join(os.path.dirname(__file__), '..', '.github', 'workflows', filename)
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def test_start_exercise_workflow():
    workflow = load_workflow('0-start-exercise.yml')

    # Check trigger condition
    # Some pyyaml parses 'on' as True.
    trigger = workflow.get('on') or workflow.get(True)
    assert trigger is not None
    assert 'push' in trigger
    assert 'branches' in trigger['push']
    assert 'main' in trigger['push']['branches']

    jobs = workflow.get('jobs', {})

    # Check start_exercise job
    assert 'start_exercise' in jobs
    start_exercise = jobs['start_exercise']
    assert 'if' in start_exercise
    assert '!github.event.repository.is_template' in start_exercise['if']
    assert 'uses' in start_exercise
    assert start_exercise['uses'].startswith('skills/exercise-toolkit/.github/workflows/start-exercise.yml')

    # Check post_next_step_content job
    assert 'post_next_step_content' in jobs
    post_next_step_content = jobs['post_next_step_content']
    assert 'needs' in post_next_step_content
    assert 'start_exercise' in post_next_step_content['needs']

    # Check for issue url env mapping (security requirement)
    assert 'env' in post_next_step_content
    assert 'ISSUE_URL' in post_next_step_content['env']
    assert post_next_step_content['env']['ISSUE_URL'] == '${{ needs.start_exercise.outputs.issue-url }}'

    # Ensure no direct injection in run steps
    steps = post_next_step_content.get('steps', [])
    for step in steps:
        if 'run' in step:
            # Check that issue-url output is not directly used in 'run'
            assert '${{ needs.start_exercise.outputs.issue-url }}' not in step['run']
