---
layout: archive
title: "Posts by 영상처리"
permalink: /categories/imgprocessing
author_profile: true
toc: true
---
{% for category in site.categories %}
  {% if category[0] == "영상처리" %}
    {% for post in category[1] %}
      {% include archive-single.html type=list %}
    {% endfor %}
  {% endif %}  
{% endfor %}  
