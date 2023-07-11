.. role:: hidden
    :class: hidden-section
.. currentmodule:: {{ module }}

{{ name | underline}}

{% if objname == "ANOMALOUS" or objname == "ONE" or objname == "Radar" or objname == "SCAN"%}
.. autoclass:: {{ name }}
    :show-inheritance:
    :members: fit, predict
{% else %}
.. autoclass:: {{ name }}
    :show-inheritance:
    :members: fit, predict, emb
{% endif %}