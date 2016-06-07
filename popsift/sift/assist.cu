#include "assist.h"

using namespace std;

ostream& operator<<( ostream& ostr, const dim3& p )
{
    ostr << "(" << p.x << "," << p.y << "," << p.z << ")";
    return ostr;
}

