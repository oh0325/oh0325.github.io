---
layout: archive
title: "Posts by Blog"
permalink: /categories/Blog
author_profile: true
---
{% for category in site.categories %}
  {% if category[0] == "Blog" %}
    {% for post in category[1] %}
      {% include archive-single.html type=list %}
    {% endfor %}
  {% endif %}  
{% endfor %}  
