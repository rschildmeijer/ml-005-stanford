
Theta1 = [4 5 6; 3 2 2; 3 3 3];
x = [1 2 2]';
a2 = zeros(3,1)

for i = 1:3
    for j = 1:3
        a2(i) = a2(i) * x(j) * Theta1(i,j);
    end;
    a2(i) = sigmoid(a2(i));
end

a2