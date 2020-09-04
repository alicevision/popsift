#include "nppi_error_strings.h"

namespace popsift
{

const char* getErrorString( NppStatus s )
{
    switch( s )
    {
    case NPP_NOT_SUPPORTED_MODE_ERROR            : return "Unsupported mode";
    
    case NPP_INVALID_HOST_POINTER_ERROR          : return "Invalid host pointer";
    case NPP_INVALID_DEVICE_POINTER_ERROR        : return "Invalid device pointer";
    case NPP_LUT_PALETTE_BITSIZE_ERROR           : return "Lut palette bitsize error";
    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR         : return "Zc mode not supported";
    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY   : return "Insufficient compute capability";
    case NPP_TEXTURE_BIND_ERROR                  : return "Texture bind error";
    case NPP_WRONG_INTERSECTION_ROI_ERROR        : return "Wrong intersection roi error";
    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR   : return "Haar classifier pixel match error";
    case NPP_MEMFREE_ERROR                       : return "Memory free error";
    case NPP_MEMSET_ERROR                        : return "Memset error";
    case NPP_MEMCPY_ERROR                        : return "Memcpy error";
    case NPP_ALIGNMENT_ERROR                     : return "Alignment error";
    case NPP_CUDA_KERNEL_EXECUTION_ERROR         : return "Cuda kernel execution error";

    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR      : return "Round mode not supported error";
    
    case NPP_QUALITY_INDEX_ERROR                 : return "Image pixels are constant for quality index";

    case NPP_RESIZE_NO_OPERATION_ERROR           : return "One of the output image dimensions is less than 1 pixel";

    case NPP_OVERFLOW_ERROR                      : return "Number overflows the upper or lower limit of the data type";
    case NPP_NOT_EVEN_STEP_ERROR                 : return "Step value is not pixel multiple";
    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR    : return "Number of levels for histogram is less than 2";
    case NPP_LUT_NUMBER_OF_LEVELS_ERROR          : return "Number of levels for LUT is less than 2";

    case NPP_CORRUPTED_DATA_ERROR                : return "Processed data is corrupted";
    case NPP_CHANNEL_ORDER_ERROR                 : return "Wrong order of the destination channels";
    case NPP_ZERO_MASK_VALUE_ERROR               : return "All values of the mask are zero";
    case NPP_QUADRANGLE_ERROR                    : return "The quadrangle is nonconvex or degenerates into triangle, line or point";
    case NPP_RECTANGLE_ERROR                     : return "Size of the rectangle region is less than or equal to 1";
    case NPP_COEFFICIENT_ERROR                   : return "Unallowable values of the transformation coefficients";

    case NPP_NUMBER_OF_CHANNELS_ERROR            : return "Bad or unsupported number of channels";
    case NPP_COI_ERROR                           : return "Channel of interest is not 1, 2, or 3";
    case NPP_DIVISOR_ERROR                       : return "Divisor is equal to zero";

    case NPP_CHANNEL_ERROR                       : return "Illegal channel index";
    case NPP_STRIDE_ERROR                        : return "Stride is less than the row length";
    
    case NPP_ANCHOR_ERROR                        : return "Anchor point is outside mask";
    case NPP_MASK_SIZE_ERROR                     : return "Lower bound is larger than upper bound";

    case NPP_RESIZE_FACTOR_ERROR                 : return "resize factor error";
    case NPP_INTERPOLATION_ERROR                 : return "interpolation error";
    case NPP_MIRROR_FLIP_ERROR                   : return "mirror flip error";
    case NPP_MOMENT_00_ZERO_ERROR                : return "moment 00 zero error";
    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR      : return "threshold negative level error";
    case NPP_THRESHOLD_ERROR                     : return "threshold error";
    case NPP_CONTEXT_MATCH_ERROR                 : return "context match error";
    case NPP_FFT_FLAG_ERROR                      : return "FFT flag error";
    case NPP_FFT_ORDER_ERROR                     : return "FFT order error";
    case NPP_STEP_ERROR                          : return "Step is less or equal zero";
    case NPP_SCALE_RANGE_ERROR                   : return "Scale range error";
    case NPP_DATA_TYPE_ERROR                     : return "Data type error";
    case NPP_OUT_OFF_RANGE_ERROR                 : return "Out of range error";
    case NPP_DIVIDE_BY_ZERO_ERROR                : return "Division by zero error";
    case NPP_MEMORY_ALLOCATION_ERR               : return "Memory allocation error";
    case NPP_NULL_POINTER_ERROR                  : return "Null pointer error";
    case NPP_RANGE_ERROR                         : return "Range error";
    case NPP_SIZE_ERROR                          : return "Size error";
    case NPP_BAD_ARGUMENT_ERROR                  : return "Bad argument error";
    case NPP_NO_MEMORY_ERROR                     : return "No memory error";
    case NPP_NOT_IMPLEMENTED_ERROR               : return "Not implemented error";
    case NPP_ERROR                               : return "Error";
    case NPP_ERROR_RESERVED                      : return "Reserved Error code";
    
    /* success */
    case NPP_SUCCESS                             : return "Success";

    /* positive return-codes indicate warnings */
    case NPP_NO_OPERATION_WARNING                : return "W: Indicates that no operation was performed";
    case NPP_DIVIDE_BY_ZERO_WARNING              : return "W: Divisor is zero however does not terminate the execution";
    case NPP_AFFINE_QUAD_INCORRECT_WARNING       : return "W: Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded.";
    case NPP_WRONG_INTERSECTION_ROI_WARNING      : return "W: The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed.";
    case NPP_WRONG_INTERSECTION_QUAD_WARNING     : return "W: The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed.";
    case NPP_DOUBLE_SIZE_WARNING                 : return "W: Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.";
    
    case NPP_MISALIGNED_DST_ROI_WARNING          : return "W: Speed reduction due to uncoalesced memory accesses warning.";
    default : return "No error explanation found";
    }
}

std::ostream& operator<<( std::ostream& ostr, NppStatus s )
{
    ostr << getErrorString(s);
    return ostr;
}

}

