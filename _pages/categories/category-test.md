---
layout: archive
title: "Posts by Test"
permalink: /categories/TEST
author_profile: true
toc: true
---
{% for category in site.categories %}
  {% if category[0] == "TEST" %}
    {% for post in category[1] %}
      {% include archive-single.html type=list %}
    {% endfor %}
  {% endif %}  
{% endfor %}  
