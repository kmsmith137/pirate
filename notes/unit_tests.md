# Randomized unit tests

Unit tests are run through a script `pirate_frb test [-n NITER]` with the following structure:
```py
for i in range(niter):  # niter specified on command line (default 100)
    test1()   # runs a test which randomizes its own parameters
    test2()   # runs a test which randomizes its own parameters
    # etc

def test1():
    """Example: Suppose we're testing a function f1(i,j,k) of 3 integer arguments."""

    # Randomize (i,j,k).
    # Details of the randomization will depend on the function being tested.
    i, j, k = np.random.randint(0, 10, size=3)

    # Subsequent code tests that f1(i,j,k) returns the correct value.
```
Many of our tests depend on a large number of parameters. For example, many functions have a DedispersionConfig argument, and a DedispersionConfig object has dozens of parameters.

**GUIDING PRINCIPLE:** In large parameter spaces, randomized testing is better than choosing a few specific cases to test. When tests fail, it can be difficult to predict in advance which cases will trigger the failure, and enumeration is impractical in large parameter spaces, so randomizing is the best strategy. Another convenience of randomization: a quick smoke test (`-n 5`) and an overnight comprehensive test (`-n 1000`) can be done with the same script.

The rest of this note is dedicated to exploring some of the nuances and design patterns that arise with randomized testing, using random DedispersionConfigs as a running example.

1. Sometimes, randomization logic is complicated, and deserves its own helper function (especially if re-used in multiple places). For example, there is a static member function DedispersionConfig::make_random() which returns a random config. As you add tests, you may find that randomization logic starts being repeated. Look for opportunities to refactor into reusable helper functions.

2. Feel free to implement randomization logic in either C++ or python -- whatever is more convenient, "feels" like good design, and leads to reasonable running time.

3. Sometimes, reusable randomization logic needs to be "tweaked" in specific tests. For example, DedispersionConfig has a `dtype` field which is randomly either float32, or float16. A few of the tests require float32, so we introduced a boolean flag `force_float32=false` in DedispersionConfig::make_random(). (Over time, make_random() has grown to have ~6 arguments of this type, that are now in their own struct DedispersionConfig::RandomArgs.)

   An alternative mechanism would be to "filter and retry", e.g.
   ```py
    # Alternative to introducing the `force_float32` flag: filter and retry
    while True:
        cfg = DedispersionConfig.make_random()
        if cfg.dtype == np.float32:
            break
   ```
   Generally speaking, I prefer the "adding arguments" pattern to the "filter and retry" pattern, but it's not a hard rule -- feel free to implement filter-and-retry if it's "cleaner" in a specific situation.

4. Randomization is often a tradeoff between good coverage (sampling all parts of a parameter space) and reasonable running time. Think carefully about how to balance this tradeoff. Often, it makes sense to randomize within some upper bound on a simple machine-independent proxy for running time.

   For example, we might randomize an array shape {m,n,p} within a hardcoded upper bound on (m*n*p), in a situation where running time is roughly proportional to array size. This specific case arises so frequently that there is a helper function for it: ksgpu::random_integers_with_bounded_product().

5. So far, we've focused on randomizing integer (or boolean) parameters that determine "shapes and sizes", but randomizing "data" arrays is also powerful, and should be done by default. This is usually more straightforward in python than in C++, but there are some C++ helper functions in ksgpu/rand_utils.hpp.

6. Think carefully about how to design randomization logic that exposes corner cases, taking into consideration the details of the code being tested. For example, suppose a test involves a length-N nonnegative-valued array `data`, and zeroes in the data array trigger special code paths. Something like this might make sense:
   ```py
    # start with a uniform random array
    data = np.random.uniform(size=N)

    # choose a random probability 0 <= p <= 1, which is sometimes 0 or 1
    p = np.clip(np.random.uniform(-0.1,2), 0, 1)

    # randomly zero data array elements with probability (1-p)
    data *= (np.random.uniform(size=N) < p)
    ```
    Feel free to get creative, and design customized randomization logic to sample corner cases, with specific code paths in mind.
    Just make sure that the main non-corner case is sampled an order-one fraction of the time, and leave a few comments so the intent is clear.

7. Avoid hardcoding specific cases. It's sometimes tempting to hardcode specific cases to test, but designing better randomization logic is preferable. For example, in the above example, we might be tempted to hardcode a corner case where the `data` array is all zeroes (if this corner case is a concern). The randomization logic above, which assigns ~5% probability to the all-zeroes case, is a better approach. (Remember that we routinely do test runs with 100 iterations or more.)

8. The `coverage` command. There is a CLI command `pirate_frb coverage` that is best explained by anecdote.

   On one occasion, we found that a particular randomized test was failing very infrequently (~1% of the time), which was undesirable since the tests needed to run for a long time to expose the bug. It turned out that the failure happened for a specific type of DedispersionConfig (dtype=float32, multiple primary trees with different max_width values), and DedispersionConfig::make_random() was inadvertently written in a way where this case was rare. This specific issue has now been fixed, but we wanted to "record" the fact that it was important to sample this case reasonably often.

   The `pirate_frb coverage` command calls randomization utility functions and reports probabilities of particular events that are important in specific unit tests. For example, it calls DedispersionConfig::make_random() and reports the probability of (dtype=float32, multiple primary trees with different max_width values). We run `pirate_frb coverage` by hand occasionally to make sure that we aren't undersampling any important case. (Currently, this is just an informal judgement, but we might make it more automated in the future.)

   If you write a unit test where a specific type of random event is important, then you may want to add code to `pirate_frb coverage` to monitor the probability, in the future as code evolves. If you find yourself sampling and inspecting the output of a randomization function, or adding code to a test to check that a particular case is sampled, this is probably a sign that you should extend `pirate_frb coverage`. (You may need to refactor your randomization logic into its own function, so that it's callable from `coverage`.)

9. We try to design tests so that running for 100 iterations is a reasonable default that tests all code paths with high probability. Sometimes the best way to do this is by putting a short "inner" loop inside the main length-niter outer loop.

   Here's a specific example that occurs often. The `pirate_frb` package includes precompiled kernels (for example it might contain N=80 "cdd2" kernels, a specific type of kernel). We have a unit test test_cdd2() which chooses a random kernel to test (as part of more general randomization logic). Our high-level test script looks like this (schematically):
   ```py
    for i in range(niter):  # niter specified on command line (default 100)
        # Ensure that each kernel is run ~10 times (in expectation value) in a full run (niter=100)
        for _ in range((N+9)//10):
            # Test one random cdd2 kernel, with randomized arguments.
            test_cdd2()
        # ...more tests follow
   ```

10. Avoid redundant or unnecessary tests. They create "friction" when refactoring, and slow down the test suite. We often run the test suite for a fixed amount of time (not a fixed number of iterations) with `pirate_frb test -t`, so unnecessary tests will make the test suite less powerful, by reducing the number of iterations.

11. An exception to the above: occasionally, a test has a small enough parameter space that we can easily enumerate and "exhaust" the parameter space. (For example, some tests have no parameters.) In this case, it makes more sense to run the entire test in the first iteration:
    ```py
    for i in range(niter):
        if i==0:
           test_with_no_params()
        randomized_test1()
        randomized_test2()
        # ...
    ```
    These cases are rare -- you should only do this if you can fully exhaust all parameters (including "data" arrays) in a short amount of compute time, or if the "test" is an informational print-statement for a human, rather than a true code test.
