"""Parts of the main window, lifted out of it.

MainWindow was one 2600-line class holding eleven distinct jobs. These modules
each take one of them and OWN its state, rather than reaching back into the
window for it — that is the difference between splitting a God object and just
moving its guts somewhere less visible. Each controller keeps a reference to the
window for the things that genuinely are shared (the current cloud, the viewer,
the panels, the status line) and nothing else.

The window still owns the signal wiring, so what connects to what stays readable
in one place; it just delegates the work here.
"""
