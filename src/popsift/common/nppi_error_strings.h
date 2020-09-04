/*
 * Copyright 2020, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <iostream>

#include <nppdefs.h>

namespace popsift
{

const char* getErrorString( NppStatus s );

std::ostream& operator<<( std::ostream& ostr, NppStatus s );

};

