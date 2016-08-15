/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "assist.h"

using namespace std;

ostream& operator<<( ostream& ostr, const dim3& p )
{
    ostr << "(" << p.x << "," << p.y << "," << p.z << ")";
    return ostr;
}

