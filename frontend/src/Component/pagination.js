import React, { Component } from 'react';
import './component.scss'

class Pagination extends Component {
    constructor(props) {
        super(props)

    }

    render() {
        const { paginationStart, paginationEnd, totalRecords, gotoFirstPage, gotoPreviousPage, currentPage, TotalPages, gotoNexttPage, gotoLastPage } = this.props;
        return (
            <div className='pagination-main'>
                <span>{`${paginationStart} to ${paginationEnd} of ${totalRecords}`}</span>
                <div>
                    <span title="First" className={`button-span ${currentPage === 1 ? 'disabled' : ''}`} onClick={gotoFirstPage}>{`<<`}</span>
                    <span title="Previous" className={`button-span ${currentPage === 1 ? 'disabled' : ''}`} onClick={gotoPreviousPage} >{`<`}</span>
                    <span>{`Page ${currentPage} of ${TotalPages}`}</span>
                    <span title="Next" className={`button-span ${currentPage === TotalPages ? 'disabled' : ''}`} onClick={gotoNexttPage}>{`>`}</span>
                    <span title="Last" className={`button-span ${currentPage === TotalPages ? 'disabled' : ''}`} onClick={gotoLastPage}>{`>>`}</span>
                </div>
            </div>
        )
    }

}

export default Pagination;