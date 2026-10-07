# Software Autofocus Plan

Ways to describe an image-based (software) autofocus plan.

`useq` does not define which routines exist, or what their settings mean: the
`method` and `settings` fields are interpreted by the acquisition engine that runs
the sequence. This keeps the schema free of any particular engine's catalog of
algorithms, while a saved sequence still travels with the exact settings it was run
with.

::: useq.SoftwareAutofocusPlan
    options:
        show_signature: true
        show_signature_annotations: true
        members:
            - as_action
            - event
            - should_autofocus

::: useq.SoftwareAxesBasedAF
    options:
        show_source: true
        show_signature: true
        show_signature_annotations: true
        members:
            - should_autofocus
