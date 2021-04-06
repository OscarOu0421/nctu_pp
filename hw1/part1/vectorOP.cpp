#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float limit = _pp_vset_float(9.999999f);
  __pp_vec_int y;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int count;
  __pp_vec_int one = _pp_vset_int(1);
  __pp_mask maskTrue, maskEqualZero, maskNotEqualZero, maskIsPositive, maskBigLimit;
  int addElement = N % VECTOR_WIDTH;

  if(addElement)
  {
    for(int i=N; i<N+VECTOR_WIDTH; i++)
    {
      values[i] = 0.f;
      exponents[i] = 1;
    }
  }

  // float *newValues;

  // /*  if N % vector width != 0 , add new memory size and set value = 0 to meet N % vecor width = 0  */
  // if(addElement)
  // {
  //   newValues = new float[N + (VECTOR_WIDTH - addElement -1)];
  //   std::copy(values, values+N, newValues);
  //   for(int i=N; i<(N+(VECTOR_WIDTH - addElement -1)); i++)
  //     newValues[i] = 0.f;
  // }
  // else
  // {
  //   newValues = new float[N];
  //   std::copy(values, values+N, newValues);
  // }

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    /*  All true  */
    maskTrue = _pp_init_ones();

    /*  All false */
    maskEqualZero = _pp_init_ones(0);

    _pp_vload_float(x, values+i, maskTrue); //x = values[i];

    _pp_vload_int(y, exponents+i, maskTrue);  //y = exponents[i];

    _pp_veq_int(maskEqualZero, y, zero, maskTrue); //if(y==0){

    _pp_vset_float(result, 1.f, maskEqualZero); //output[i] = 1.f;

    maskNotEqualZero = _pp_mask_not(maskEqualZero); // } else{

    _pp_vmove_float(result, x, maskNotEqualZero);  //result = x;

    _pp_vsub_int(count, y, one, maskNotEqualZero);  //count = y - 1;

    /*  All false */
    maskIsPositive = _pp_init_ones(0);

    _pp_vgt_int(maskIsPositive, count, zero, maskTrue);

    while(_pp_cntbits(maskIsPositive))  //while(count > 0){
    {
      _pp_vmult_float(result, result, x, maskIsPositive); //result *= x;

      _pp_vsub_int(count, count, one, maskIsPositive);  //count--;

      _pp_vgt_int(maskIsPositive, count, zero, maskTrue); //}
    }

    /*  All false */
    maskBigLimit = _pp_init_ones(0);

    _pp_vgt_float(maskBigLimit, result, limit, maskTrue); //if (result > 9.999999f){

    _pp_vset_float(result, 9.999999f, maskBigLimit);  //result = 9.999999f;}

    _pp_vstore_float(output + i, result, maskTrue); //output[i] = result;
  }
  
  // delete [] newValues;
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float sum =0.0;
  __pp_vec_float sumV = _pp_vset_float(0.f);
  __pp_mask maskTrue;

    /*  All true  */
  maskTrue = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    _pp_vload_float(sumV, values+i, maskTrue);

    for(int j = 1; j < VECTOR_WIDTH; j*=2)
    {
      __pp_vec_float temp;
      _pp_interleave_float(temp, sumV);
      _pp_hadd_float(sumV, temp);
    }
    
    sum += sumV.value[0];
  }

  return sum;
}