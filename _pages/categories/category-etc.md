---
layout: archive
title: "Posts by Another Things"
permalink: /categories/etc
author_profile: true
toc: true
---
{% for category in site.categories %}
  {% if category[0] == "Another Things" %}
    {% for post in category[1] %}
      {% include archive-single.html type=list %}
    {% endfor %}
  {% endif %}  
{% endfor %}
