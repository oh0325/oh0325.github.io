---
layout: archive
title: "Posts by 그 외"
permalink: /categories/etc
author_profile: true
toc: true
---
{% for category in site.categories %}
  {% if category[0] == "그 외" %}
    {% for post in category[1] %}
      {% include archive-single.html type=list %}
    {% endfor %}
  {% endif %}  
{% endfor %}
