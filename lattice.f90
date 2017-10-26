module lattice
implicit none
contains

function genpoly(n)
    implicit none
    integer::n,c,m,i,j
    integer,allocatable::seed(:),seed2(:)
    integer::genpoly(n) ! 0 = north, 1 = east, 2 = south, 3 = west
    real *8::r(n)

    call init_random_seed()
    call random_number(r)
    do i=1,n
        genpoly(i) = int(4*r(i))
    enddo

endfunction genpoly

function end2end(chain)
    implicit none
    integer::chain(:)
    real *8::end2end
    integer::n,x,i,y

    n = size(chain)
    x = 0
    y = 0
    do i=1,n
        if(chain(i)==0)then
            y = y+1
        elseif(chain(i)==1)then
            x = x+1
        elseif(chain(i)==2)then
            y = y-1
        elseif(chain(i)==3)then
            x = x-1
        endif
    enddo

    end2end = sqrt(1.0d0*(x**2+y**2))

endfunction end2end


subroutine init_random_seed() !Not my code! from user Francesco on Stackoverflow
    INTEGER :: i, n, clock
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed
    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))
    CALL SYSTEM_CLOCK(COUNT=clock)
    seed = clock + 37 * (/ (i - 1, i = 1, n)/)
    CALL RANDOM_SEED(PUT = seed)
    DEALLOCATE(seed)
endsubroutine init_random_seed

endmodule lattice
