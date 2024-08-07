"""
This module defines the protocol for
[`Trainer`][flight.learning.trainers.base.Trainer] objects.

A `Trainer` is responsible for facilitating the training/fitting, testing,
and evaluation of train objects that implement
[`HasParameters`][flight.learning.modules.prototypes.HasParameters].
Users are able to use one of the `Trainer` implementations provided by Flight.
Additionally, for more specific use cases, users can implement their own
`Trainer` object by simply implementing the `Trainer` protocol.
"""
