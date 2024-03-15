import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import constant from '../../../../Constants/validator'

const apiServices = new APIServices();

class Cards extends Component {
    constructor(props) {
        super();
        this.state = {
            cardData: [],
            loading: false,
        };
    }

    componentWillReceiveProps = (props) => {
        this.getCardData(props)
    }

    getCardData = (props) => {
        var self = this;
        const { startDate, endDate, regionId, countryId, cityId } = props;
        self.setState({ loading: true, cardData: [] })
        apiServices.getDemoBottomCards(startDate, endDate, regionId, countryId, cityId).then((data) => {
            self.setState({ loading: false })
            if (data) {
                self.setState({
                    cardData: data
                })
            }
        });
    }

    render() {

        return (
            <div className='cards-main' >
                {this.state.cardData.length > 0 ?
                    this.state.cardData.map((data, i) =>
                        <div id="card_data" className="cards" >
                            <div className='card-top'>
                                <div className="count_top"> {data.Name} </div>
                                <div className="count" id={data.Name}> {window.numberFormat(data.CY)} </div>
                            </div>
                            <div className='indicator-bottom' >
                                <span style={{ flex: '1' }}>{window.numberFormat(data.LY)} LY</span>
                                <div className='indicator-2'>
                                    <i className={`${constant.cardsArrowIndicator(window.numberFormat(data.VLY))}`}></i>
                                    <span>{window.numberFormat(data.VLY)}% VLY</span>
                                </div>
                            </div>
                            <div className='indicator-bottom' >
                                <span style={{ flex: '1' }}>{window.numberFormat(data.Budget)} BGT</span>
                                <div className='indicator-2'>
                                    <i className={`${constant.cardsArrowIndicator(window.numberFormat(data.VTG))}`}></i>
                                    <span >{window.numberFormat(data.VTG)}% VTG</span>
                                </div>
                            </div>
                        </div>)
                    : <div />}
            </div>
        )
    }
}

export default Cards;
