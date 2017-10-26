module lattice
implicit none
contains

function genpoly(n)
    implicit none
    integer::n,c,i
    integer::seed
    integer, dimension(n)::genpoly ! 0 = north, 1 = east, 2 = south, 3 = west
    real *8,dimension()::r
    call random_seed(size = n)
    call random_seed(put=seed)
    deallocate(seed)

    call random_number(r)

    do i=1,n
        genpoly(i) = int(4*r(i))


    enddo

endfunction genpoly

endmodule lattice
