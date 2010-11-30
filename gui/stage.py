#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: gatoatigrado (nicholas tung) [ntung at ntung]
# Copyright 2010 University of California, Berkeley

# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .
from __future__ import division, print_function
from collections import namedtuple, defaultdict
import sys; reload(sys); sys.setdefaultencoding('utf-8') # print unicode
from gatoatigrado_lib import (ExecuteIn, Path, SubProc, dict, get_singleton,
    list, memoize_file, persistent_var, pprint, process_jinja2, set, sort_asc,
    sort_desc)
