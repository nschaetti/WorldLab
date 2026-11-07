"""
Copyright (C) 2025 boardGPT Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

class ActivationRecorder:
    """
    Records activations
    """

    def __init__(self):
        """
        Constructor
        """
        self.records = {}  # end def __init__
    # end def __init__

    @property
    def keys(self):
        """
        Keys
        """
        return list(self.records.keys())  # end def keys
    # end keys

    def clear(self):
        """
        Clears records
        :return:
        """
        self.records = {}  # end def clear
    # end def clear

    def save(
            self,
            name,
            tensor  # end def save
    ):
        """
        Save a tensor

        Args:
            name: name of the tensor
            tensor: tensor
        """
        # detach pour éviter d'accumuler des gradients
        self.records[name] = tensor.detach().cpu()
    # end def save

    def get(
            self,
            name  # end def get
    ):
        """
        Get a tensor by name.

        Args:
            name: name of the tensor

        Returns:
            tensor: tensor
        """
        return self.records.get(name, None)
    # end def get

    def all(self):
        """
        Get all tensors
        """
        return self.records  # end def all
    # end def all
# end class ActivationRecorder
# end class ActivationRecorder
