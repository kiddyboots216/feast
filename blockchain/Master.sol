pragma solidity 0.4.21;

import "Query.sol";

contract Master {
    address public owner;
    // address public target = 0xB7e25827fB83d5Ec795276C5343518B047D32199;
    uint256 public id;
    mapping(address => uint256[]) public queries;
    mapping(uint256 => address) private addressBook;

    address q;

    event QueryCreated(address,address);

    function Master() public {
        owner = msg.sender;
    }

    function query(address target)
        public
        payable
    {
        int[] memory lst = new int[](3);
        lst[0] = 0;
        lst[1] = 1;
        lst[2] = 2;
        q = address(new Query(3, lst, 0, 0, 5, lst, 3));
        address[] memory addrlst = new address[](1);
        addrlst[0] = target;
        pingClients(addrlst);
    }

    function pingClients(address[] clientList) internal {
        uint clientLen = clientList.length;
        for (uint i = 0; i < clientLen; i++) {
            emit QueryCreated(clientList[i], q);
        }
    }
}
