/-  ls=lagoon
/+  *test,
    *lagoon,
    math,
    saloon
::
::::
  ::
^|
|_  $:  atol=_.1e-3          :: absolute tolerance for precision of operations
        rtol=_.1e-5          :: relative tolerance for precision of operations
    ==
::  Auxiliary tools
++  expect-near
  |=  [expected=@rs actual=@rs]  ^-  tang
  ?:  (~(is-close rs:math [%z rtol]) `@rs`expected `@rs`actual)
    ~
  :~  [%palm [": " ~ ~ ~] [leaf+"expected" "{<expected>}"]]
      [%palm [": " ~ ~ ~] [leaf+"actual" "{<actual>}"]]
  ==
++  expect-near-per
  |=  [expected=@rs actual=@rs]  ^-  tang
  =/  pdif  (div:rs:math (abs:rs:math (sub:rs:math actual expected)) expected)
  ?:  (lth:rs:math pdif atol)
    ~
  :~  [%palm [": " ~ ~ ~] [leaf+"expected" "{<expected>}"]]
      [%palm [": " ~ ~ ~] [leaf+"actual" "{<actual>}"]]
  ==
++  expect-ray-near
  |=  [expected=ray:ls actual=ray:ls]  ^-  tang
  ::  element-wise comparison with tolerance for FP rounding
  =/  exp-els  (ravel:la expected)
  =/  act-els  (ravel:la actual)
  |-  ^-  tang
  ?~  exp-els  ~
  ?~  act-els  ~
  ?.  (~(is-close rd:math [%z .~1e-9]) i.exp-els i.act-els)
    :~  [%palm [": " ~ ~ ~] [leaf+"expected" "{<expected>}"]]
        [%palm [": " ~ ~ ~] [leaf+"actual" "{<actual>}"]]
    ==
  $(exp-els t.exp-els, act-els t.act-els)
  
::  Comparison
::++  test-isclose  !!
::++  test-allclose  !!
++  test-factorial  ^-  tang
  ;:  weld
    %+  expect-near
      .1
      (factorial:rs:math .0)
    %+  expect-near
      .1
      (factorial:rs:math .1)
    %+  expect-near
      .120
      (factorial:rs:math .5)
    ==
++  test-abs  ^-  tang
  ;:  weld
    %+  expect-near
      .1
      (abs:rs:math .-1)
    %+  expect-near
      .1
      (abs:rs:math .1)
    %+  expect-near
      .120
      (abs:rs:math .-120)
    ==
++  test-exp  ^-  tang
  ;:  weld
    %+  expect-near-per
      .148.413
      (exp:rs:math .5)
    %+  expect-near-per
      .26903.186
      (exp:rs:math .10.2)
    ==
++  test-eml  ^-  tang
  ;:  weld
    %+  expect-near
      .1
      (eml:rs:math .0 .1)
    %+  expect-near-per
      (sub:rs:math e:rs:math .1)
      (eml:rs:math .1 e:rs:math)
    ==
::  exercise the ray lift on @rd values
++  test-eml-ray  ^-  tang
  =/  x  (en-ray:la [~[2] 6 %i754 ~] ~[.~0 .~1])
  =/  y  (en-ray:la [~[2] 6 %i754 ~] ~[.~1 e:rd:math])
  =/  expected
    (en-ray:la [~[2] 6 %i754 ~] ~[.~1 (sub:rd:math e:rd:math .~1)])
  (expect-ray-near expected (eml:sa:saloon x y))
++  test-pow-n  ^-  tang
  ;:  weld
    %+  expect-near-per
      .132.651
      (pow-n:rs:math .5.1 .3)
    %+  expect-near-per
      .-27
      (pow-n:rs:math .-3 .3)
    ==
::  log-2 is log base 2
++  test-log-2  ^-  tang
  ;:  weld
    %+  expect-near-per
      .4.9069
      (log-2:rs:math .30)
    %+  expect-near-per
      .-0.8625
      (log-2:rs:math .0.55)
    ==
++  test-log  ^-  tang
  ;:  weld
    %+  expect-near-per
      .-2.30259
      (log:rs:math .0.1)
    %+  expect-near-per
      .4.094345
      (log:rs:math .60)
    ==
::  log-10 is log base 10
++  test-log-10  ^-  tang
  ;:  weld
    %+  expect-near-per
      .1.477121
      (log-10:rs:math .30)
    %+  expect-near-per
      .-0.180456
      (log-10:rs:math .0.66)
    ==
++  test-pow  ^-  tang
  ;:  weld
    %+  expect-near-per
      .202.582
      (pow:rs:math .5 .3.3)
    %+  expect-near-per
      .-391.35393
      (pow:rs:math .-3.3 .5)
    %+  expect-near-per
      .0.00493627
      (pow:rs:math .5 .-3.3)
    %+  expect-near-per
      .391.35393
      (pow:rs:math .3.3 .5)
    ==
++  test-sqrt  ^-  tang
  ;:  weld
    %+  expect-near-per
      .2
      (sqrt:rs:math .4)
    %+  expect-near-per
      .1.41421
      (sqrt:rs:math .2)
    ==
++  test-cbrt  ^-  tang
  ;:  weld
    %+  expect-near-per
      .3
      (cbrt:rs:math .27)
    %+  expect-near-per
      .1.63864
      (cbrt:rs:math .4.4)
    ==
::++  test-binomial  !!
++  test-sin  ^-  tang
  ;:  weld
    %+  expect-near-per
      .-0.756802
      (sin:rs:math .4)
    %+  expect-near-per
      .0.522687
      (sin:rs:math .0.55)
    ==
++  test-cos  ^-  tang
  ;:  weld
    %+  expect-near-per
      .-0.65364
      (cos:rs:math .4)
    %+  expect-near-per
      .0.852525
      (cos:rs:math .0.55)
    ==
++  test-tan  ^-  tang
  ;:  weld
    %+  expect-near-per
      .1.15782
      (tan:rs:math .4)
    %+  expect-near-per
      .0.613105
      (tan:rs:math .0.55)
    ==
::  csc, sec, cot not yet exposed in saloon — these live in saloon-old
::  TODO: re-add when reciprocal trig functions are lifted
::
++  test-asin  ^-  tang
  ;:  weld
    %+  expect-near-per
      .1.11977
      (asin:rs:math .0.9)
    %+  expect-near-per
      .0.55
      (asin:rs:math .0.522687)
    ==
++  test-acos  ^-  tang
  ;:  weld
    %+  expect-near-per
      .2.28318
      (acos:rs:math .-0.65364)
    %+  expect-near-per
      .0.55
      (acos:rs:math .0.852525)
    ==
++  test-atan  ^-  tang
  ;:  weld
    %+  expect-near-per
      .0.732815
      (atan:rs:math .0.9)
    %+  expect-near-per
      .0.55
      (atan:rs:math .0.613105)
    ==
::  Transformer activation function tests (ray-level, @rs)
::
++  test-relu  ^-  tang
  =/  x  (en-ray:la [~[4] 5 %i754 ~] ~[.-2 .-0.5 .0.5 .3])
  =/  expected  (en-ray:la [~[4] 5 %i754 ~] ~[.0 .0 .0.5 .3])
  (expect-ray-near expected (relu:sa:saloon x))
++  test-sigmoid  ^-  tang
  =/  x  (en-ray:la [~[2] 5 %i754 ~] ~[.0 .1])
  ::  sigmoid(0) = 0.5, sigmoid(1) ~ 0.7310586
  =/  expected  (en-ray:la [~[2] 5 %i754 ~] ~[.0.5 .0.7310586])
  (expect-ray-near expected (sigmoid:sa:saloon x))
++  test-tanh  ^-  tang
  =/  x  (en-ray:la [~[2] 5 %i754 ~] ~[.0 .1])
  ::  tanh(0) = 0, tanh(1) ~ 0.7615942
  =/  expected  (en-ray:la [~[2] 5 %i754 ~] ~[.0 .0.7615942])
  (expect-ray-near expected (tanh:sa:saloon x))
++  test-softmax  ^-  tang
  =/  x  (en-ray:la [~[3] 5 %i754 ~] ~[.1 .2 .3])
  ::  softmax([1,2,3]) ~ [0.0900306, 0.2447285, 0.6652409]
  =/  expected  (en-ray:la [~[3] 5 %i754 ~] ~[.0.0900306 .0.2447285 .0.6652409])
  (expect-ray-near expected (softmax:sa:saloon x))
::
::: https://en.wikipedia.org/wiki/Particular_values_of_the_gamma_function
::++  test-gamma  ^-  tang
  ::;:  weld
    ::%+  expect-near
      ::.1.772453850
      ::(gamma:rs:math .0.5)
    ::%+  expect-near
      ::.0.886226925
      ::(gamma:rs:math .1.5)
    ::%+  expect-near
      ::.1.329340388
      ::(gamma:rs:math .2.5)
    ::%+  expect-near
      ::.3.32335097
      ::(gamma:rs:math .3.5)
    ::%+  expect-near
      ::.-3.544907701
      ::(gamma:rs:math .-0.5)
    ::==
--
