name: Bug report
description: Create a report to help us improve
title: "[BUG]: "
labels: [triage needed, bug]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to report a bug in the Cosdata project.
  - type: markdown
    attributes:
      value: |
        Before filing a new issue, **please do a quick search** to check that it hasn't already been filed on the issue tracker. You can do this by going to [this link](https://github.com/cosdata/cosdata/issues) and typing some words related to the issue in the search box next to the "New issue" button.
  - type: textarea
    id: issue-description
    attributes:
      label: Describe the bug
      description: A concise description of what issue you're experiencing. **Example:** "ADD _EXAMPLE" // TODO
    validations:
      required: true
  - type: textarea
    id: steps-to-reproduce
    attributes:
      label: Steps To Reproduce
      description: Steps to reproduce the behavior. **Example:** "ADD _EXAMPLE" // TODO
    validations:
      required: true
  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected Behavior
      description: A clear and concise description of what you expected to happen. **Example:** "ADD _EXAMPLE" // TODO
    validations:
      required: true
  - type: textarea
    id: screenshots-or-videos
    attributes:
      label: Screenshots/Videos
      description: |
        If applicable, add screenshots or videos to help explain your problem.

        **Tip:** You can attach images or log files by clicking this area to highlight it and then dragging files in.
    validations:
      required: false
  - type: input
    id: device
    attributes:
      label: What device/machine are you using?
      description: Please specify the device/machine you're using.
      placeholder: Mac OS
    validations:
      required: false
  - type: input
    id: cosdata-version
    attributes:
      label: Which version of the Cosdata are you using?
      description: // TODO - how to get version
      placeholder: v0.1.0-beta
    validations:
      required: false
  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Add any other context about the problem here.
    validations:
      required: false
