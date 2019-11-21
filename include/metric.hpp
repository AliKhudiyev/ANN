
#pragma once

struct Metric{
    double accuracy, precision;
    double sensitivity, specificity;

    double accuracy() const;
    double sensitivity() const;
    double precision() const;
    double specificity() const;
};
