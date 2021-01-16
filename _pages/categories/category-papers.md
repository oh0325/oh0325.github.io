---
layout: archive
title: "Posts by Papers"
permalink: /categories/papers
author_profile: true
toc: true
---
{% for category in site.categories %}
  {% if category[0] == "Papers" %}
    {% for post in category[1] %}
      {% include archive-single.html type=list %}
    {% endfor %}
  {% endif %}  
{% endfor %}  
